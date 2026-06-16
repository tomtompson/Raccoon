from types import SimpleNamespace

from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import Paragraph
from reportlab.platypus.tables import Table

from raccoon.report.StaticRetrieverReport import StaticRetrieverReport


def _retriever(
    name: str,
    ndcg: float,
    recall: float,
    corpus: dict | None = None,
    config: dict | None = None,
):
    return SimpleNamespace(
        retriever_type=name,
        config=config or {"max_length": 256, "score_function": "dot"},
        corpus=corpus or {},
        queries={
            "q1": "first query",
            "q2": {"text": "second query"},
        },
        results={
            "q1": {"d1": 1.0, "d2": 0.5},
            "q2": {"d2": 0.9},
        },
        retrieval_metrics={"NDCG@10": ndcg, "Recall@10": recall},
        metrics={
            "index_time": {"indexing": {"time_in_seconds": 0.2}},
            "query_time": {"search": {"time_in_seconds": 0.1}},
        },
        reranker=None,
        rerank_results={},
        rerank_metrics={},
        rerank_retrieval_metrics={},
        topk=20,
    )


def _paragraph_text(flowable):
    return flowable.getPlainText() if isinstance(flowable, Paragraph) else ""


def _table_texts(table):
    texts = []
    for row in getattr(table, "_cellvalues", []):
        for cell in row:
            if isinstance(cell, Paragraph):
                texts.append(cell.getPlainText())
            else:
                texts.append(str(cell))
    return texts


def _drawing_texts(drawing):
    return [item.text for item in drawing.contents if isinstance(item, String)]


def test_static_report_builds_comparison_pdf(tmp_path):
    corpus = {
        "d1": {"title": "Document One", "text": "alpha beta gamma"},
        "d2": {"text": "delta epsilon"},
    }
    retrievers = [
        _retriever("bm25", 0.72, 0.64, corpus),
        _retriever("dense", 0.81, 0.7),
    ]
    report = StaticRetrieverReport()

    corpus_rows = dict(report._corpus_rows(retrievers))
    assert corpus_rows["Documents"] == 2
    assert corpus_rows["Documents with title"] == 1
    assert corpus_rows["Queries"] == 2

    config_rows = dict(report._config_rows(retrievers[0], {"config_rows": 20}))
    assert config_rows["max_length"] == 256
    assert config_rows["topk"] == 20

    metric_maps = report._metric_maps(retrievers, "retrieval_metrics", {"retrieval_metric_rows": 10})
    assert report._selected_metrics(metric_maps, ["NDCG@10", "Recall@10", "Missing@1"]) == [
        "NDCG@10",
        "Recall@10",
    ]

    path = report.generate_report(
        "Retriever Test Report",
        retrievers,
        output_path=tmp_path / "report.pdf",
        config={"comparison_metric": ["NDCG@10", "Recall@10"]},
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_static_report_compares_multiple_retrievers_and_explains_metrics(tmp_path):
    corpus = {
        "d1": {"title": "Document One", "text": "alpha beta gamma"},
        "d2": {"title": "Document Two", "text": "delta epsilon"},
        "d3": {"text": "zeta eta theta"},
    }
    retrievers = [
        _retriever("bm25", 0.61, 0.58, corpus, {"language": "dutch"}),
        _retriever("dense", 0.72, 0.63, config={"model_id": "test-model", "max_length": 206}),
        _retriever("hybrid", 0.78, 0.71, config={"rrf_k": 60}),
        _retriever("linear", 0.69, 0.66, config={"dense_candidate_k": 500, "graph_rrf_weight": 1.0}),
    ]
    report = StaticRetrieverReport()
    config = {"comparison_metric": ["NDCG@10", "Recall@10"], "retrieval_metric_rows": 20}
    styles = report._styles()

    intro = report._intro(config, len(retrievers), styles)
    intro_text = " ".join(item.getPlainText() for item in intro if isinstance(item, Paragraph))
    assert "retrieval-augmented generation" in intro_text
    assert "context window" in intro_text

    metric_maps = report._metric_maps(retrievers, "retrieval_metrics", config)
    assert [name for name, _ in metric_maps] == [
        "SimpleNamespace (bm25)",
        "SimpleNamespace (dense)",
        "SimpleNamespace (hybrid)",
        "SimpleNamespace (linear)",
    ]
    assert all({"NDCG@10", "Recall@10"} <= set(metrics) for _, metrics in metric_maps)

    comparison = report._metric_comparison("Retrieval Metrics", metric_maps, ["NDCG@10", "Recall@10"], styles)
    note_text = " ".join(item.getPlainText() for item in comparison if isinstance(item, Paragraph))
    assert "Metric guide" in note_text
    assert "RAG pipeline" in note_text
    assert "Recall@k" in note_text
    assert "NDCG@k" in note_text

    path = report.generate_report(
        "Multiple Retriever Report",
        retrievers,
        output_path=tmp_path / "multi_report.pdf",
        config=config,
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_static_report_includes_reranker_sections(tmp_path):
    corpus = {
        "d1": {"title": "Document One", "text": "alpha beta gamma"},
        "d2": {"title": "Document Two", "text": "delta epsilon"},
    }
    retrievers = [
        _retriever("dense", 0.72, 0.63, corpus, {"model_id": "dense-model", "max_length": 206}),
        _retriever("hybrid", 0.78, 0.71, corpus, {"rrf_k": 60}),
    ]
    for index, retriever in enumerate(retrievers, start=1):
        retriever.reranker = SimpleNamespace(
            model_id="BAAI/bge-reranker-v2-m3",
            top_k=10,
            batch_size=4,
            device="cpu",
        )
        retriever.rerank_results = {
            "q1": {"d2": 2.5 + index, "d1": 1.2},
            "q2": {"d2": 2.2 + index},
        }
        retriever.rerank_metrics = {
            "total_wall_time_sec": 0.45 + index / 10,
            "avg_time_query_sec": 0.12,
            "pairs_per_sec": 18.5,
        }
        retriever.rerank_retrieval_metrics = {
            "NDCG@10": 0.8 + index / 100,
            "Recall@10": 0.74 + index / 100,
        }

    report = StaticRetrieverReport()
    config = {"comparison_metric": ["NDCG@10", "Recall@10"], "rerank_comparison_metric": ["NDCG@10"]}
    styles = report._styles()

    reranker_rows = dict(report._reranker_rows(retrievers[0]))
    assert reranker_rows == {
        "enabled": True,
        "model_id": "BAAI/bge-reranker-v2-m3",
        "top_k": 10,
        "batch_size": 4,
        "device": "cpu",
    }

    assert report._rerank_values(retrievers, "total_wall_time_sec") == [
        ("SimpleNamespace (dense)", 0.55),
        ("SimpleNamespace (hybrid)", 0.65),
    ]

    rerank_metric_maps = report._metric_maps(retrievers, "rerank_retrieval_metrics", config)
    assert report._selected_metrics(rerank_metric_maps, config["rerank_comparison_metric"]) == ["NDCG@10"]

    rerank_comparison = report._metric_comparison("Rerank Retrieval Metrics", rerank_metric_maps, ["NDCG@10"], styles)
    assert rerank_comparison
    assert not any(
        isinstance(item, Paragraph) and "Metric guide" in item.getPlainText()
        for item in rerank_comparison
    )

    path = report.generate_report(
        "Reranker Report",
        retrievers,
        output_path=tmp_path / "reranker_report.pdf",
        config=config,
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_static_report_builds_complete_report_with_quality_speed_legend(tmp_path):
    corpus = {
        "d1": {"title": "Document One", "text": "alpha beta gamma"},
        "d2": {"title": "Document Two", "text": "delta epsilon"},
        "d3": {"title": "Document Three", "text": "zeta eta theta"},
    }
    retrievers = [
        _retriever("bm25", 0.61, 0.58, corpus, {"max_length": 256, "score_function": "bm25"}),
        _retriever("dense", 0.76, 0.69, corpus, {"model_id": "dense-model", "max_length": 512}),
    ]

    for index, retriever in enumerate(retrievers, start=1):
        rerank_retrieval_metrics = {
            "NDCG@10": 0.8 + index / 100,
            "Recall@10": 0.74 + index / 100,
        }
        retriever.retrieval_metrics = {
            retriever.retriever_type: {
                "NDCG@10": retriever.retrieval_metrics["NDCG@10"],
                "Recall@10": retriever.retrieval_metrics["Recall@10"],
            },
            "rerank": rerank_retrieval_metrics,
        }
        retriever.metrics = {
            "index_time": {
                "indexing": {
                    "time_in_seconds": 0.8 + index / 10,
                    "storage_size_mb": 12.5 + index,
                }
            },
            "query_time": {
                "search": {
                    "time_in_seconds": 0.20 + index / 100,
                    "time_per_query_in_seconds": 0.10 + index / 100,
                    "queries_per_second": 9.0 - index,
                }
            },
        }
        retriever.reranker = SimpleNamespace(
            retriever_type="cross-encoder",
            model_id="BAAI/bge-reranker-v2-m3",
            top_k=10,
            batch_size=4,
            device="cpu",
        )
        retriever.rerank_results = {
            "q1": {"d1": 2.4 + index, "d2": 1.1},
            "q2": {"d2": 2.2 + index, "d3": 1.0},
        }
        retriever.rerank_metrics = {
            "total_wall_time_sec": 0.30 + index / 10,
            "avg_time_query_sec": 0.05 + index / 100,
            "pairs_per_sec": 18.5,
        }
        retriever.rerank_retrieval_metrics = rerank_retrieval_metrics

    retrievers[0].query_graph_path = "raccoon/report/images/logo.png"
    retrievers[0].query_graph_query_id = "q1"
    retrievers[0].query_graph_query = "first query"

    qrels = {
        "q1": {"d1": 1, "d3": 0},
        "q2": {"d2": 1},
    }
    config = {
        "comparison_metric": ["NDCG@10", "Recall@10"],
        "rerank_comparison_metric": ["NDCG@10", "Recall@10"],
        "sample_queries": 2,
        "sample_results_per_query": 2,
        "sample_text_chars": 80,
        "query_graph_doc_rows": 2,
        "query_graph_doc_chars": 40,
        "scalability_target_docs": 1000,
        "scalability_target_queries": 100,
    }

    report = StaticRetrieverReport()
    report.qrels = qrels
    styles = report._styles()
    story = report._story(
        title="Complete Retriever Report",
        ds_description="Synthetic dataset description",
        retrievers=retrievers,
        styles=styles,
        config=config,
        n_samples=2,
        results_per_query=2,
        sample_chars=80,
    )

    paragraph_texts = [_paragraph_text(item) for item in story if isinstance(item, Paragraph)]
    for expected in (
        "Complete Retriever Report",
        "Synthetic dataset description",
        "Corpus Summary",
        "Retriever Comparison",
        "Query Graph - SimpleNamespace (bm25)",
        "Quality versus Query Throughput",
        "Rank Distribution",
        "Scalability Estimates",
        "Sample Top Results",
        "Sample Top Rerank Results",
    ):
        assert expected in paragraph_texts

    quality_speed_values = report._quality_speed_values(retrievers, config)
    quality_speed_names = {name for name, _qps, _recall in quality_speed_values}
    assert quality_speed_names == {
        report._name(retriever)
        for retriever in retrievers
    } | {
        report._reranked_name(retriever)
        for retriever in retrievers
    }

    quality_heading = next(
        index
        for index, item in enumerate(story)
        if isinstance(item, Paragraph)
        and _paragraph_text(item) == "Quality versus Query Throughput"
    )
    quality_legend = next(item for item in story[quality_heading + 1 :] if isinstance(item, Table))
    assert quality_speed_names <= set(_table_texts(quality_legend))

    quality_speed_chart_texts = next(
        texts
        for texts in (_drawing_texts(item) for item in story if isinstance(item, Drawing))
        if "Recall@10" in texts and "Queries/sec" in texts
    )
    assert quality_speed_names.isdisjoint(quality_speed_chart_texts)

    path = report.generate_report(
        title="Complete Retriever Report",
        ds_description="Synthetic dataset description",
        retrievers=retrievers,
        qrels=qrels,
        output_path=tmp_path / "complete_report.pdf",
        config=config,
        n_samples=2,
        language="english",
    )

    assert path.exists()
    assert path.stat().st_size > 0
