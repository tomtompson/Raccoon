from __future__ import annotations

import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class StaticRetrieverReport:
    def generate_report(
        self,
        title: str | None = None,
        retrievers: Any | None = None,
        output_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        n_samples: int | None = None,
    ) -> Path:
        config = config or {}
        path = self._path(output_path)
        styles = self._styles()
        retrievers = self._list(retrievers)

        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=36,
            bottomMargin=36,
        )
        doc.build(
            self._story(
                title=title or "Retriever Report",
                retrievers=retrievers,
                styles=styles,
                config=config,
                n_samples=self._int(n_samples or config.get("sample_queries", 3), 3),
                results_per_query=self._int(config.get("sample_results_per_query", 3), 3),
                sample_chars=self._int(config.get("sample_text_chars", 220), 220),
            )
        )
        print(f"Saved report to: {path}")
        return path


    def _story(
        self,
        title: str,
        retrievers: list[Any],
        styles: dict[str, Any],
        config: dict[str, Any],
        n_samples: int,
        results_per_query: int,
        sample_chars: int,
    ) -> list[Any]:
        story: list[Any] = []
        logo = Path("raccoon/report/images/logo.png")
        if logo.exists():
            story += [Image(str(logo), width=22 * mm, height=22 * mm), Spacer(1, 4)]

        story += [
            Paragraph(self._esc(title), styles["Title"]),
            Paragraph(datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"), styles["MutedCenter"]),
            Spacer(1, 14),
        ]

        if not retrievers:
            story.append(Paragraph("No retriever results were provided.", styles["Body"]))
            return story

        story += self._comparison(retrievers, styles, config)

        for retriever in retrievers:
            story += self._retriever_section(
                retriever=retriever,
                styles=styles,
                config=config,
                n_samples=n_samples,
                results_per_query=results_per_query,
                sample_chars=sample_chars,
            )
            story.append(Spacer(1, 12))
        return story

    def _retriever_section(
        self,
        retriever: Any,
        styles: dict[str, Any],
        config: dict[str, Any],
        n_samples: int,
        results_per_query: int,
        sample_chars: int,
    ) -> list[Any]:
        results = self._results(getattr(retriever, "results", {}))
        rerank_results = self._results(getattr(retriever, "rerank_results", {}))
        metrics = getattr(retriever, "metrics", {}) or {}
        retrieval_metrics = getattr(retriever, "retrieval_metrics", None)
        rerank_metrics = getattr(retriever, "rerank_metrics", {}) or {}
        rerank_retrieval_metrics = getattr(retriever, "rerank_retrieval_metrics", None)

        story = [Paragraph(self._esc(self._name(retriever)), styles["Section"])]
        story += self._table("Result Summary", self._summary(results), styles)
        story += self._table("Retrieval Metrics", self._metric_rows(retrieval_metrics, config), styles)
        story += self._table("Retriever Metrics", self._flatten(metrics)[: self._int(config.get("metric_rows", 12), 12)], styles)
        story += self._table("Reranker", self._reranker_rows(retriever), styles)
        story += self._table("Rerank Metrics", self._flatten(rerank_metrics), styles)
        story += self._table("Rerank Retrieval Metrics", self._metric_rows(rerank_retrieval_metrics, config), styles)
        story += self._samples("Sample Top Results", retriever, results, styles, n_samples, results_per_query, sample_chars)
        story += self._samples("Sample Top Rerank Results", retriever, rerank_results, styles, n_samples, results_per_query, sample_chars)
        return story

    def _table(self, title: str, rows: list[tuple[str, Any]], styles: dict[str, Any]) -> list[Any]:
        if not rows:
            return []

        table = Table(
            [
                [
                    Paragraph(self._esc(self._label(label)), styles["Cell"]),
                    Paragraph(self._esc(self._format(value)), styles["Cell"]),
                ]
                for label, value in rows
            ],
            colWidths=[48 * mm, 122 * mm],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return [Paragraph(self._esc(title), styles["Subsection"]), table, Spacer(1, 8)]

    def _comparison(self, retrievers: list[Any], styles: dict[str, Any], config: dict[str, Any]) -> list[Any]:
        if len(retrievers) < 2:
            return []

        story = [Paragraph("Retriever Comparison", styles["Section"])]

        metric_maps = [
            (self._name(retriever), dict(self._metric_rows(getattr(retriever, "retrieval_metrics", None), config)))
            for retriever in retrievers
        ]
        metric_names = sorted({name for _, metrics in metric_maps for name in metrics})
        if metric_names:
            chart_metric = config.get("comparison_metric") or metric_names[0]
            chart_values = [(name, metrics[chart_metric]) for name, metrics in metric_maps if chart_metric in metrics]
            story += [
                Paragraph(self._esc(f"{chart_metric} comparison"), styles["Subsection"]),
                self._bar_chart(chart_values),
                Spacer(1, 8),
            ]

        rerank_metric_maps = [
            (self._name(retriever), dict(self._metric_rows(getattr(retriever, "rerank_retrieval_metrics", None), config)))
            for retriever in retrievers
        ]
        rerank_metric_names = sorted({name for _, metrics in rerank_metric_maps for name in metrics})
        if rerank_metric_names:
            chart_metric = config.get("rerank_comparison_metric") or config.get("comparison_metric") or rerank_metric_names[0]
            chart_values = [(name, metrics[chart_metric]) for name, metrics in rerank_metric_maps if chart_metric in metrics]
            if chart_values:
                story += [
                    Paragraph(self._esc(f"Reranked {chart_metric} comparison"), styles["Subsection"]),
                    self._bar_chart(chart_values),
                    Spacer(1, 8),
                ]

        index_values = self._runtime_values(retrievers, "index_time")
        query_values = self._runtime_values(retrievers, "query_time")
        if index_values:
            story += [
                Paragraph("Index time comparison (seconds)", styles["Subsection"]),
                self._bar_chart(index_values),
                Spacer(1, 8),
            ]
        if query_values:
            story += [
                Paragraph("Query time comparison (seconds)", styles["Subsection"]),
                self._bar_chart(query_values),
                Spacer(1, 8),
            ]

        rerank_values = self._rerank_values(retrievers, "total_wall_time_sec")
        if rerank_values:
            story += [
                Paragraph("Rerank time comparison (seconds)", styles["Subsection"]),
                self._bar_chart(rerank_values),
                Spacer(1, 8),
            ]

        return story + [Spacer(1, 4)] if len(story) > 1 else []

    def _runtime_values(self, retrievers: list[Any], metric_key: str) -> list[tuple[str, float]]:
        values = []
        for retriever in retrievers:
            seconds = self._time_in_seconds((getattr(retriever, "metrics", {}) or {}).get(metric_key))
            if seconds is not None:
                values.append((self._name(retriever), seconds))
        return values

    def _time_in_seconds(self, value: Any) -> float | None:
        if not isinstance(value, dict):
            return self._number(value)
        if "time_in_seconds" in value:
            return self._number(value["time_in_seconds"])
        for child in value.values():
            seconds = self._time_in_seconds(child)
            if seconds is not None:
                return seconds
        return None

    def _rerank_values(self, retrievers: list[Any], metric_key: str) -> list[tuple[str, float]]:
        values = []
        for retriever in retrievers:
            metrics = getattr(retriever, "rerank_metrics", {}) or {}
            value = self._number(metrics.get(metric_key))
            if value is not None:
                values.append((self._name(retriever), value))
        return values

    def _bar_chart(self, values: list[tuple[str, float]]) -> Drawing:
        width = 170 * mm
        row_height = 15
        label_width = 50 * mm
        bar_width = 90 * mm
        height = max(24, 10 + row_height * len(values))
        drawing = Drawing(width, height)
        max_value = max([value for _, value in values] + [1])

        y = height - 14
        for name, value in values:
            drawing.add(String(0, y, self._truncate(name, 24), fontSize=7, fillColor=colors.HexColor("#111827")))
            drawing.add(Rect(label_width, y - 3, bar_width, 6, fillColor=colors.HexColor("#E5E7EB"), strokeColor=None))
            drawing.add(Rect(label_width, y - 3, bar_width * (value / max_value), 6, fillColor=colors.HexColor("#2563EB"), strokeColor=None))
            drawing.add(String(label_width + bar_width + 5, y, self._format(value), fontSize=7, fillColor=colors.HexColor("#111827")))
            y -= row_height
        return drawing

    def _samples(
        self,
        title: str,
        retriever: Any,
        results: list[tuple[str, list[tuple[str, float | None]]]],
        styles: dict[str, Any],
        n_samples: int,
        results_per_query: int,
        sample_chars: int,
    ) -> list[Any]:
        if not results:
            return []

        queries = getattr(retriever, "queries", {}) or {}
        corpus = getattr(retriever, "corpus", {}) or {}
        story: list[Any] = [Paragraph(title, styles["Subsection"])]

        for query_id, hits in results[:n_samples]:
            story.append(
                Paragraph(
                    f"<b>Query {self._esc(query_id)}:</b> {self._esc(self._get_query_text(queries, query_id))}",
                    styles["Body"],
                )
            )
            if not hits:
                story += [Paragraph("No results.", styles["Muted"]), Spacer(1, 5)]
                continue

            for rank, (doc_id, score) in enumerate(hits[:results_per_query], start=1):
                full_text = self._get_text(corpus, doc_id)
                sample = self._truncate(full_text, sample_chars)
                score_text = f" | score {self._format(score)}" if score is not None else ""
                story += [
                    Paragraph(f"<b>{rank}. Document {self._esc(doc_id)}{self._esc(score_text)}</b>", styles["Small"]),
                    Paragraph(self._esc(sample), styles["Small"]),
                ]
            story.append(Spacer(1, 7))
        return story

    def _results(self, results: Any) -> list[tuple[str, list[tuple[str, float | None]]]]:
        if not isinstance(results, dict):
            return []
        return [(str(query_id), self._hits(row)) for query_id, row in results.items()]

    def _hits(self, row: Any) -> list[tuple[str, float | None]]:
        if isinstance(row, dict) and isinstance(row.get("hits"), list):
            row = row["hits"]
        if isinstance(row, dict):
            return [(str(doc_id), self._number(score)) for doc_id, score in row.items()]
        if not isinstance(row, list):
            return []

        hits = []
        for index, hit in enumerate(row, start=1):
            if isinstance(hit, dict):
                doc_id = hit.get("document_id") or hit.get("doc_id") or hit.get("id") or hit.get("_id") or f"hit-{index}"
                hits.append((str(doc_id), self._number(hit.get("score", hit.get("_score")))))
            elif isinstance(hit, (list, tuple)) and len(hit) >= 2:
                hits.append((str(hit[0]), self._number(hit[1])))
            else:
                hits.append((str(hit), None))
        return hits

    def _summary(self, results: list[tuple[str, list[tuple[str, float | None]]]]) -> list[tuple[str, Any]]:
        counts = [len(hits) for _, hits in results]
        scores = [score for _, hits in results for _, score in hits if score is not None]
        total = sum(counts)
        query_count = len(results)

        rows: list[tuple[str, Any]] = [
            ("Queries", query_count),
            ("Total results", total),
            ("Avg results/query", total / query_count if query_count else 0),
            ("Min results/query", min(counts) if counts else 0),
            ("Max results/query", max(counts) if counts else 0),
        ]
        if scores:
            rows += [("Min score", min(scores)), ("Avg score", mean(scores)), ("Max score", max(scores))]
        return rows

    def _metric_rows(self, metrics: Any, config: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
        if not metrics:
            return []

        if isinstance(metrics, (list, tuple)) and len(metrics) == 4 and all(isinstance(item, dict) for item in metrics):
            rows: list[tuple[str, Any]] = []
            for name, values in zip(("NDCG", "MAP", "Recall", "Precision"), metrics):
                rows += self._metric_rows(values, config, name)
            return self._sort_metrics(rows, config)

        if not isinstance(metrics, dict):
            return []

        rows = []
        for key, value in metrics.get("summary", metrics).items():
            label = self._join_metric(prefix, str(key))
            if isinstance(value, dict):
                rows += self._metric_rows(value, config, label)
            elif self._number(value) is not None:
                rows.append((self._metric_name(label), self._number(value)))
        return self._sort_metrics(rows, config)

    def _sort_metrics(self, rows: list[tuple[str, Any]], config: dict[str, Any]) -> list[tuple[str, Any]]:
        order = {"NDCG": 0, "MAP": 1, "Recall": 2, "Precision": 3, "MRR": 4}

        def key(row: tuple[str, Any]) -> tuple[int, int, str]:
            metric, _, raw_k = row[0].partition("@")
            digits = "".join(ch for ch in raw_k if ch.isdigit())
            return (order.get(metric, len(order)), int(digits or 0), row[0])

        return sorted(rows, key=key)[: self._int(config.get("retrieval_metric_rows", 24), 24)]

    def _flatten(self, value: Any, prefix: str = "") -> list[tuple[str, Any]]:
        if not isinstance(value, dict):
            return []
        rows = []
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                rows += self._flatten(item, label)
            elif isinstance(item, (str, int, float, bool)) or item is None:
                rows.append((label, item))
        return rows

    def _reranker_rows(self, retriever: Any) -> list[tuple[str, Any]]:
        reranker = getattr(retriever, "reranker", None)
        if reranker is None:
            return []
        return [
            ("enabled", True),
            ("model_id", getattr(reranker, "model_id", "")),
            ("top_k", getattr(reranker, "top_k", "")),
            ("batch_size", getattr(reranker, "batch_size", "")),
            ("device", getattr(reranker, "device", "")),
        ]

    def _get_query_text(self, queries: dict[Any, Any], query_id: str) -> str:
        query = self._lookup(queries, query_id)
        if isinstance(query, dict):
            return str(query.get("text") or query.get("query") or query.get("query_text") or query_id)
        return str(query or query_id)

    def _get_text(self, corpus: dict[Any, Any], doc_id: str) -> str:
        doc = self._lookup(corpus, doc_id)
        if isinstance(doc, str):
            return doc
        if not isinstance(doc, dict):
            return "(document text unavailable)"

        title = str(doc.get("title") or doc.get("name") or "").strip()
        text = str(doc.get("text") or doc.get("content") or doc.get("page_content") or doc.get("body") or "").strip()
        return f"{title}: {text}" if title and text else title or text or "(document text unavailable)"

    def _lookup(self, values: dict[Any, Any], key: str) -> Any:
        if not isinstance(values, dict):
            return None
        if key in values:
            return values[key]
        try:
            return values.get(int(key))
        except (TypeError, ValueError):
            return values.get(str(key))

    def _styles(self) -> dict[str, Any]:
        styles = getSampleStyleSheet()
        styles["Title"].alignment = TA_CENTER
        styles.add(styles["Normal"].clone("MutedCenter", alignment=TA_CENTER, fontSize=9, textColor=colors.HexColor("#6B7280")))
        styles.add(styles["Heading2"].clone("Section", fontSize=12, leading=15, spaceAfter=7, backColor=colors.HexColor("#F3F4F6"), borderPadding=5))
        styles.add(styles["Heading3"].clone("Subsection", fontSize=10, leading=12, spaceBefore=6, spaceAfter=5))
        styles.add(styles["BodyText"].clone("Body", fontSize=9, leading=11, spaceAfter=4))
        styles.add(styles["BodyText"].clone("Small", fontSize=8, leading=10, spaceAfter=2))
        styles.add(styles["BodyText"].clone("Muted", fontSize=8, leading=10, textColor=colors.HexColor("#6B7280")))
        styles.add(styles["BodyText"].clone("Cell", fontSize=8, leading=10))
        return styles

    def _path(self, output_path: str | Path | None) -> Path:
        path = Path(output_path or "./reports/report.pdf")
        if path.suffix.lower() != ".pdf":
            path = path / "report.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _list(self, value: Any | None) -> list[Any]:
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple, set)) else [value]

    def _name(self, retriever: Any) -> str:
        retriever_type = getattr(retriever, "retriever_type", None)
        name = type(retriever).__name__
        return f"{name} ({retriever_type})" if retriever_type else name

    def _join_metric(self, prefix: str, label: str) -> str:
        if not prefix or "@" in label:
            return label
        return f"{prefix}@{label}" if label.isdigit() else f"{prefix}.{label}"

    def _metric_name(self, label: str) -> str:
        metric, sep, k = label.partition("@")
        metric = {
            "ndcg": "NDCG",
            "map": "MAP",
            "mrr": "MRR",
            "p": "Precision",
            "precision": "Precision",
            "recall": "Recall",
        }.get(metric.lower(), metric)
        return f"{metric}@{k}" if sep else metric

    def _number(self, value: Any) -> float | None:
        if isinstance(value, dict):
            value = value.get("score", value.get("_score"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return self._truncate(str(value), 70)

    def _label(self, value: str) -> str:
        return str(value).replace("_", " ").replace(".", " / ").title()

    def _truncate(self, text: str, limit: int) -> str:
        text = " ".join(str(text or "").split())
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    def _int(self, value: Any, default: int) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def _esc(self, value: Any) -> str:
        return escape(str(value or ""))
