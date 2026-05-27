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
import platform
import torch
import psutil
import os
import random

class StaticRetrieverReport:
    def generate_report(
        self,
        title: str | None = None,
        ds_description: str | None = None,
        retrievers: Any | None = None,
        output_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        n_samples: int | None = None,
        language: str = "english",
    ) -> Path:
        config = config or {}
        path = self._path(output_path)
        styles = self._styles()
        retrievers = self._list(retrievers)
        if language.lower() in ("english", "en"):
            from .util.EnglishText import EnglishText
            self.text_class = EnglishText()
        elif language.lower() in ("dutch", "nl", "nederlands"):
            from .util.DutchText import DutchText
            self.text_class = DutchText()

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
                ds_description=ds_description,
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
        ds_description: str | None = None,
    ) -> list[Any]:
        story: list[Any] = []
        logo = Path("raccoon/report/images/logo.png")
        if logo.exists():
            story += [Image(str(logo), width=50 * mm, height=50 * mm)]

        story += [
            Paragraph(self._esc(title), styles["Title"]),
            Paragraph(datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"), styles["MutedCenter"]),
            Spacer(1, 6),
            *self._intro(config, len(retrievers), styles),
        ]

        if ds_description:
            story += [
                Paragraph(self._esc(ds_description.replace('"', '')), styles["Body"]),
                Spacer(1, 6),
            ]

        story += self._machine_summary(styles)

        if not retrievers:
            story.append(Paragraph("No retriever results were provided.", styles["Body"]))
            return story

        story += self._corpus_summary(retrievers, styles)
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
        retrieval_metrics = getattr(retriever, "retrieval_metrics", {}) or {}
        rerank_metrics = getattr(retriever, "rerank_metrics", {}) or {}
        rerank_retrieval_metrics = getattr(retriever, "retrieval_metrics", None).get("rerank")

        retrieval_metrics = getattr(retriever, "retrieval_metrics", {}) or {}
        base_key = getattr(retriever, "retriever_type", None)
        base_retrieval_metrics = retrieval_metrics.get(base_key, retrieval_metrics)
        rerank_retrieval_metrics = retrieval_metrics.get("rerank", {})

        headers, rows = self._metric_table(
            {base_key or "retrieval": base_retrieval_metrics},
            config,
        )

        rerank_headers, rerank_rows = self._metric_table(
            {"rerank": rerank_retrieval_metrics},
            config,
        )

        story = [Paragraph(self._esc(self._name(retriever)), styles["Section"])]
        story += self._table("Configuration", self._config_rows(retriever, config), styles)
        story += self._table("Result Summary", self._summary(results), styles)
        story += self._table("Retrieval Metrics", rows, styles, headers=headers)
        story += self._table("Retriever Metrics", self._flatten(metrics)[: self._int(config.get("metric_rows", 20), 20)], styles)
        story += self._table("Reranker", self._reranker_rows(retriever), styles)
        story += self._table("Rerank Metrics", self._flatten(rerank_metrics), styles)
        story += self._table("Rerank Retrieval Metrics", rerank_rows, styles, headers=rerank_headers)
        story += self._samples("Sample Top Results", retriever, results, styles, n_samples, results_per_query, sample_chars)
        story += self._samples("Sample Top Rerank Results", retriever, rerank_results, styles, n_samples, results_per_query, sample_chars)
        return story

    def _table(
    self,
    title: str,
    rows: list[Any],
    styles: dict[str, Any],
    headers: list[str] | None = None,
    ) -> list[Any]:
        if not rows:
            return []

        if headers:
            data = [
                [
                    Paragraph(self._esc(self._label(header)), styles["Cell"])
                    for header in headers
                ]
            ]

            data += [
                [
                    Paragraph(self._esc(self._format(value)), styles["Cell"])
                    for value in row
                ]
                for row in rows
            ]

            col_count = len(headers)
            page_width = 170 * mm
            col_widths = [page_width / col_count] * col_count

            style_commands = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]

        else:
            data = [
                [
                    Paragraph(self._esc(self._label(label)), styles["Cell"]),
                    Paragraph(self._esc(self._format(value)), styles["Cell"]),
                ]
                for label, value in rows
            ]

            col_widths = [48 * mm, 122 * mm]

            style_commands = [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]

        table = Table(data, colWidths=col_widths, hAlign="LEFT")
        table.setStyle(TableStyle(style_commands))

        return [Paragraph(self._esc(title), styles["Subsection"]), table, Spacer(1, 8)]

    def _corpus_summary(self, retrievers: list[Any], styles: dict[str, Any]) -> list[Any]:
        rows = self._corpus_rows(retrievers)
        raw_queries: dict = getattr(retrievers[0], "queries", {}) or {}
        queries = random.sample(list(raw_queries.values()), min(len(raw_queries), 3)) if raw_queries else []
        if not rows:
            return []
        return ([Paragraph("Corpus Summary", styles["Section"])] + self._table("Corpus and Queries", rows, styles) + 
        [Paragraph("Sample Query", styles["Subsection"])] 
        + [Paragraph(self._esc(self._query_text(query)), styles["Body"]) for query in queries]) + [Spacer(1, 16)]

    def _machine_summary(self, styles: dict[str, Any]) -> list[Any]:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

        rows = [
            ("System", platform.system()),
            ("Processor", platform.processor()),
            ("Cores", os.cpu_count()),
            ("RAM (GB)", ram_gb),
            ("Python version", platform.python_version()),
        ]

        if torch.cuda.is_available():
            rows.extend([
                ("CUDA", f"Available ({torch.cuda.device_count()} GPU(s))"),
                ("GPU Name", torch.cuda.get_device_name(0)),
            ])
        else:
            rows.append(("CUDA", "No CUDA available"))
            
        return [Paragraph("Machine Summary", styles["Section"])] + self._table("Machine Summary", rows, styles,)

    def _intro(self, config: dict[str, Any], retriever_count: int, styles: dict[str, Any]) -> list[Any]:
        retriever_word = "retriever" if retriever_count == 1 else "retrievers"
        intro = self.text_class.intro_text.format(retriever_count=retriever_count, retriever_word=retriever_word)
        if not self.text_class.intro_text:
            return []
        return [Paragraph(self._esc(intro), styles["Body"]), Spacer(1, 8)]

    def _corpus_rows(self, retrievers: list[Any]) -> list[tuple[str, Any]]:
        source = next(
            (
                retriever
                for retriever in retrievers
                if isinstance(getattr(retriever, "corpus", None), dict)
                and bool(getattr(retriever, "corpus", None))
            ),
            None,
        ) or next(
            (
                retriever
                for retriever in retrievers
                if isinstance(getattr(retriever, "queries", None), dict)
                and bool(getattr(retriever, "queries", None))
            ),
            None,
        )
        if source is None:
            return []

        corpus = getattr(source, "corpus", {}) or {}
        queries = getattr(source, "queries", {}) or {}
        rows: list[tuple[str, Any]] = []

        if isinstance(corpus, dict) and corpus:
            texts = [self._doc_text(doc) for doc in corpus.values()]
            lengths = [len(text) for text in texts]
            titled = sum(1 for doc in corpus.values() if isinstance(doc, dict) and str(doc.get("title") or "").strip())
            rows += [
                ("Documents", len(corpus)),
                ("Documents with title", titled),
                ("Avg document chars", mean(lengths) if lengths else 0),
                ("Min document chars", min(lengths) if lengths else 0),
                ("Max document chars", max(lengths) if lengths else 0),
            ]

        if isinstance(queries, dict) and queries:
            query_texts = [self._query_text(query, str(query_id)) for query_id, query in queries.items()]
            query_lengths = [len(text) for text in query_texts]
            rows += [
                ("Queries", len(queries)),
                ("Avg query chars", mean(query_lengths) if query_lengths else 0),
                ("Min query chars", min(query_lengths) if query_lengths else 0),
                ("Max query chars", max(query_lengths) if query_lengths else 0),
            ]

        return rows

    def _comparison(self, retrievers: list[Any], styles: dict[str, Any], config: dict[str, Any]) -> list[Any]:
        if len(retrievers) < 2:
            return []

        story = [Paragraph("Retriever Comparison", styles["Section"])]

        metric_maps = self._comparison_metric_maps_with_rerank(retrievers, config)
        chart_metrics = self._selected_metrics(metric_maps, config.get("comparison_metric"))

        if chart_metrics:

            story += [
            Paragraph("Retrieval and Rerank Metrics", styles["Subsection"]),
            self._comparison_table(metric_maps, chart_metrics, styles),
            Spacer(1, 6),]
            story += self._comparison_color_table(styles)
            story += [
            self._overlap_grouped_bar_chart(metric_maps, chart_metrics),
            Spacer(1, 6),]

        story += [
            *self._metric_explanation("Retrieval Metrics", styles),
            Spacer(1, 110),
        ]
        
        practical_rows = self._practical_limit_rows(retrievers, config)
        if practical_rows:
            story += [
                Paragraph("Practical Limitations and Runtime Trade-offs", styles["Subsection"]),
                self._practical_limit_table(practical_rows, styles),
                Spacer(1, 10),
            ]

            quality_speed_values = self._quality_speed_values(retrievers, config)
            if quality_speed_values:
                story += [
                    Paragraph("Quality versus Query Throughput", styles["Subsection"]),
                    self._quality_speed_chart(quality_speed_values),
                ]
            
                story += [Paragraph(self.text_class.throughput_text, styles["Note"]), Spacer(1, 8)]
                story += [Paragraph(self.text_class.throughput_interpretation_text, styles["Note"]), Spacer(1, 8)]

        index_values = self._runtime_values(retrievers, "index_time")
        query_values = self._runtime_values(retrievers, "query_time")

        if index_values:
            story += [
                Paragraph(f"Total Index Time (seconds) ({len(retrievers[0].corpus)} docs)", styles["Subsection"]),
                self._bar_chart(index_values),
                Spacer(1, 10),
            ]

        if query_values:
            story += [
                Paragraph(f"Total Query Time (seconds) ({len(retrievers[0].queries)} queries)", styles["Subsection"]),
                self._bar_chart(query_values),
                Spacer(1, 10),
            ]

        rerank_values = self._rerank_values(retrievers, "total_wall_time_sec")
        if rerank_values:
            story += [
                Paragraph(f"Rerank Time (seconds) ({retrievers[0].reranker.top_k} - docs / {len(retrievers[0].queries)} - queries )", styles["Subsection"]),
                self._bar_chart(rerank_values),
                Spacer(1, 8),
            ]

        return story + [Spacer(1, 4)] if len(story) > 1 else []

    def _practical_limit_rows(
        self,
        retrievers: list[Any],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for retriever in retrievers:
            name = self._name(retriever)
            retriever_type = getattr(retriever, "retriever_type", "")

            retrieval_metrics = getattr(retriever, "retrieval_metrics", {}) or {}
            base_metrics = retrieval_metrics.get(retriever_type, {}) if retriever_type else {}
            metric_rows = dict(self._metric_rows(base_metrics, config))

            ndcg_10 = self._number(metric_rows.get("NDCG@10"))
            recall_10 = self._number(metric_rows.get("Recall@10"))

            metrics = getattr(retriever, "metrics", {}) or {}
            index_time = self._time_in_seconds(metrics.get("index_time"))
            query_time = self._time_in_seconds(metrics.get("query_time"))

            query_search = (
                metrics.get("query_time", {}).get("search", {})
                if isinstance(metrics.get("query_time"), dict)
                else {}
            )

            qps = self._number(query_search.get("queries_per_second"))
            time_per_query = self._number(query_search.get("time_per_query_in_seconds"))
            storage_mb = self._storage_mb(metrics)

            corpus_count = len(getattr(retriever, "corpus", {}) or {})
            dps = corpus_count / index_time if index_time and corpus_count else None

            rows.append(
                {
                    "name": name,
                    "ndcg_10": ndcg_10,
                    "recall_10": recall_10,
                    "index_time": index_time,
                    "query_time": query_time,
                    "time_per_query": time_per_query,
                    "qps": qps,
                    "storage_mb": storage_mb,
                    "dps": dps,
                }
            )

        return rows
    def _practical_limit_table(
        self,
        rows: list[dict[str, Any]],
        styles: dict[str, Any],
    ) -> Table:
        headers = [
            "Retriever",
            "Query Time",
            "Queries/Sec",
            "Index Time",
            "Document/Sec",
            "Storage MB",
        ]

        data = [
            [Paragraph(header, styles["HeaderCell"]) for header in headers]
        ]

        for row in rows:
            data.append(
                [
                    Paragraph(self._esc(row["name"]), styles["Cell"]),
                    Paragraph(self._esc(self._format_seconds(row["query_time"])), styles["Cell"]),
                    Paragraph(self._esc(self._format(row["qps"])), styles["Cell"]),
                    Paragraph(self._esc(self._format_seconds(row["index_time"])), styles["Cell"]),
                    Paragraph(self._esc(self._format(row["dps"])), styles["Cell"]),
                    Paragraph(self._esc(self._format(row["storage_mb"])), styles["Cell"]),
                ]
            )

        table = Table(
            data,
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        return table

    def _quality_speed_values(
        self,
        retrievers: list[Any],
        config: dict[str, Any],
    ) -> list[tuple[str, float, float]]:
        values = []

        for row in self._practical_limit_rows(retrievers, config):
            qps = self._number(row.get("qps"))
            recall = self._number(row.get("recall_10"))

            if qps is not None and recall is not None:
                values.append((row["name"], qps, recall))

        return values

    def _quality_speed_chart(self, values: list[tuple[str, float, float]]) -> Drawing:
        width = 100 * mm
        height = 80 * mm
        left = 5 * mm
        bottom = 5 * mm
        plot_width = 170 * mm
        plot_height = 70 * mm

        drawing = Drawing(width, height)

        max_qps = max([qps for _, qps, _ in values] + [1.0])
        max_recall = max([recall for _, _, recall in values] + [1.0])


        drawing.add(String(-20, 20 + plot_height, "Recall@10", fontSize=7,fontName="Helvetica-Bold"))
        drawing.add(String(475, -5, "Queries/sec", fontSize=7,fontName="Helvetica-Bold"))

        drawing.add(Rect(left, bottom, plot_width, plot_height, fillColor=None, strokeColor=colors.HexColor("#D1D5DB")))
        # Y axis labels (Recall)
        for i in range(6):
            value = max_recall * (i / 5)
            y = bottom + plot_height * (i / 5)

            drawing.add(
                String(
                    left - 18,
                    y - 2,
                    f"{value:.2f}",
                    fontSize=6,
                    fillColor=colors.HexColor("#374151"),
                )
            )

            drawing.add(
                Rect(
                    left - 2,
                    y,
                    2,
                    0.3,
                    fillColor=colors.HexColor("#9CA3AF"),
                    strokeColor=None,
                )
            )


        # X axis labels (Queries/sec)
        for i in range(6):
            value = max_qps * (i / 5)
            x = left + plot_width * (i / 5)

            drawing.add(
                String(
                    x - 5,
                    bottom - 10,
                    f"{value:.1f}",
                    fontSize=6,
                    fillColor=colors.HexColor("#374151"),
                )
            )

            drawing.add(
                Rect(
                    x,
                    bottom - 2,
                    0.3,
                    2,
                    fillColor=colors.HexColor("#9CA3AF"),
                    strokeColor=None,
                )
            )

        palette = [
            colors.HexColor("#2563EB"),
            colors.HexColor("#059669"),
            colors.HexColor("#D97706"),
            colors.HexColor("#7C3AED"),
            colors.HexColor("#DC2626"),
            colors.HexColor("#0891B2"),
        ]

        for index, (name, qps, recall) in enumerate(values):
            x = left + plot_width * (qps / max_qps)
            y = bottom + plot_height * (recall / max_recall)

            drawing.add(
                Rect(
                    x - 2,
                    y - 2,
                    4,
                    4,
                    fillColor=palette[index % len(palette)],
                    strokeColor=None,
                )
            )
            drawing.add(
                String(
                    x + 4,
                    y - 2,
                    self._truncate(name, 24),
                    fontSize=6,
                    fillColor=colors.HexColor("#111827"),
                    fontName="Helvetica-Bold",
                )
            )

        return drawing
    

    def _comparison_metric_maps_with_rerank(
        self,
        retrievers: list[Any],
        config: dict[str, Any],
    ) -> list[tuple[str, dict[str, float]]]:
        rows: list[tuple[str, dict[str, float]]] = []

        for retriever in retrievers:
            name = self._name(retriever)
            retrieval_metrics = getattr(retriever, "retrieval_metrics", {}) or {}

            base_key = getattr(retriever, "retriever_type", None)
            base_metrics = retrieval_metrics.get(base_key, {}) if base_key else {}
            rerank_metrics = retrieval_metrics.get("rerank", {})

            base_rows = dict(self._metric_rows(base_metrics, config))
            rerank_rows = dict(self._metric_rows(rerank_metrics, config))

            if base_rows:
                rows.append((name, base_rows))

            if rerank_rows:
                rows.append((f"{name} + rerank", rerank_rows))

        return rows


    def _comparison_color_table(self, styles: dict[str, Any]) -> list[Any]:
        rows = [
            ("Base retrieval", "#2563EB"),
            ("Rerank", "#F97316"),
        ]

        table = Table(
            [
                [
                    Paragraph("Series", styles["HeaderCell"]),
                ],
                *[
                    [
                        Paragraph(self._esc(label), styles["Cell"]),
                    ]
                    for label, hex_value in rows
                ],
            ],
            colWidths=[25 * mm, 25 * mm],
            hAlign="LEFT",
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#2563EB")),
                    ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#F97316")),
                    ("TEXTCOLOR", (0, 1), (0, 2), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        return [Paragraph("Chart Colors", styles["Subsection"]), table, Spacer(1, 8)]


    def _overlap_grouped_bar_chart(
        self,
        metric_maps: list[tuple[str, dict[str, float]]],
        metrics: list[str],
    ) -> Drawing:
        width = 170 * mm
        label_width = 48 * mm
        metric_width = 25 * mm
        bar_start = label_width + metric_width
        bar_width = 72 * mm
        row_height = 9
        group_height = 11 + row_height * len(metrics)

        base_rows = [
            (name, values)
            for name, values in metric_maps
            if not name.endswith(" + rerank")
        ]

        height = max(32, 8 + group_height * len(base_rows))
        drawing = Drawing(width, height)

        max_value = max(
            [float(values[metric]) for _, values in metric_maps for metric in metrics if metric in values]
            + [1.0]
        )

        base_color = colors.HexColor("#2563EB")
        rerank_color = colors.HexColor("#F97316")
        background_color = colors.HexColor("#E5E7EB")

        lookup = dict(metric_maps)

        y = height - 12
        for base_name, base_values in base_rows:
            rerank_name = f"{base_name} + rerank"
            rerank_values = lookup.get(rerank_name, {})

            drawing.add(
                String(
                    0,
                    y,
                    base_name,
                    fontSize=7,
                    fillColor=colors.HexColor("#111827"),
                )
            )

            for index, metric in enumerate(metrics):
                metric_y = y - 10 - row_height * index

                drawing.add(
                    String(
                        label_width,
                        metric_y + 1,
                        self._truncate(metric, 14),
                        fontSize=6,
                        fillColor=colors.HexColor("#374151"),
                    )
                )

                drawing.add(
                    Rect(
                        bar_start,
                        metric_y,
                        bar_width,
                        5,
                        fillColor=background_color,
                        strokeColor=None,
                    )
                )

                bars = []

                base_value = self._number(base_values.get(metric))
                if base_value is not None:
                    bars.append(("base", base_value, base_color))

                rerank_value = self._number(rerank_values.get(metric))
                if rerank_value is not None:
                    bars.append(("rerank", rerank_value, rerank_color))

                # Draw higher value first, so it stays visually in the back.
                # Draw lower value last, so it appears in front.
                bars = sorted(bars, key=lambda item: item[1], reverse=True)

                for _, value, color in bars:
                    fill_width = bar_width * max(0.0, value / max_value)
                    drawing.add(
                        Rect(
                            bar_start,
                            metric_y,
                            fill_width,
                            5,
                            fillColor=color,
                            strokeColor=None,
                        )
                    )

                label_parts = []
                if base_value is not None:
                    label_parts.append(f"B {self._format(base_value)}")
                if rerank_value is not None:
                    label_parts.append(f"R {self._format(rerank_value)}")

                drawing.add(
                    String(
                        bar_start + bar_width + 4,
                        metric_y + 1,
                        " / ".join(label_parts),
                        fontSize=6,
                        fillColor=colors.HexColor("#111827"),
                    )
                )

            y -= group_height

        return drawing

    def _selected_metrics(
        self,
        metric_maps: list[tuple[str, dict[str, float]]],
        requested: Any,
    ) -> list[str]:
        available = self._ordered_metric_names({name for _, metrics in metric_maps for name in metrics})
        if not available:
            return []

        selected = self._as_list(requested)
        if not selected:
            selected = [metric for metric in ("NDCG@10", "Recall@10") if metric in available]
        if not selected:
            selected = available[:4]

        return [metric for metric in selected if metric in available]

    def _metric_explanation(self, title: str, styles: dict[str, Any]) -> list[Any]:
        if title != "Retrieval Metrics":
            return []

        story = [
            Paragraph(self.text_class.metric_guide_text, styles["Note"]),
            Paragraph(self.text_class.metric_text, styles["Note"]),
        ]

        practical_text = getattr(self.text_class, "practical_interpretation_text", "")
        if practical_text:
            story.append(Paragraph(practical_text, styles["Note"]))

        story.append(Spacer(1, 4))
        return story

    def _comparison_table(
        self,
        metric_maps: list[tuple[str, dict[str, float]]],
        metrics: list[str],
        styles: dict[str, Any],
    ) -> Table:
        metric_count = max(1, len(metrics))
        table = Table(
            [
                [Paragraph("Retriever", styles["HeaderCell"])]
                + [Paragraph(self._esc(metric), styles["HeaderCell"]) for metric in metrics],
                *[
                    [Paragraph(self._esc(name), styles["Cell"])]
                    + [Paragraph(self._esc(self._format(values.get(metric))), styles["Cell"]) for metric in metrics]
                    for name, values in metric_maps
                    if any(metric in values for metric in metrics)
                ],
            ],
            colWidths=[58 * mm] + [(112 / metric_count) * mm for _ in metrics],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F3F4F6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def _grouped_bar_chart(self, metric_maps: list[tuple[str, dict[str, float]]], metrics: list[str]) -> Drawing:
        width = 170 * mm
        label_width = 46 * mm
        metric_width = 25 * mm
        bar_start = label_width + metric_width
        bar_width = 72 * mm
        row_height = 9
        group_height = 11 + row_height * len(metrics)
        rows = [(name, values) for name, values in metric_maps if any(metric in values for metric in metrics)]
        height = max(32, 8 + group_height * len(rows))
        drawing = Drawing(width, height)
        max_value = max(
            [float(values[metric]) for _, values in rows for metric in metrics if metric in values]
            + [1.0]
        )
        palette = [
            colors.HexColor("#2563EB"),
            colors.HexColor("#059669"),
            colors.HexColor("#D97706"),
            colors.HexColor("#7C3AED"),
            colors.HexColor("#DC2626"),
            colors.HexColor("#0891B2"),
        ]

        y = height - 12
        for name, values in rows:
            drawing.add(String(0, y, name, fontSize=7, fillColor=colors.HexColor("#111827")))
            for index, metric in enumerate(metrics):
                value = self._number(values.get(metric))
                metric_y = y - 10 - row_height * index
                drawing.add(String(label_width, metric_y + 1, self._truncate(metric, 14), fontSize=6, fillColor=colors.HexColor("#374151")))
                drawing.add(Rect(bar_start, metric_y, bar_width, 5, fillColor=colors.HexColor("#E5E7EB"), strokeColor=None))
                if value is not None:
                    fill_width = bar_width * max(0.0, value / max_value)
                    drawing.add(Rect(bar_start, metric_y, fill_width, 5, fillColor=palette[index % len(palette)], strokeColor=None))
                    drawing.add(
                        String(
                            bar_start + bar_width + 4,
                            metric_y + 1,
                            self._format(value),
                            fontSize=6,
                            fillColor=colors.HexColor("#111827"),
                        )
                    )
            y -= group_height
        return drawing

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

    def _storage_mb(self, metrics: dict[str, Any]) -> float | None:
        flat = dict(self._flatten(metrics))

        for key in (
            "index_time.indexing.size_in_mb",
            "index_time.encoding.size_in_mb",
            "index_time.indexing.size_mb",
            "index_time.encoding.size_mb",
        ):
            value = self._number(flat.get(key))
            if value is not None:
                return value

        for key in (
            "index_time.indexing.size_in_bytes",
            "index_time.encoding.size_in_bytes",
        ):
            value = self._number(flat.get(key))
            if value is not None:
                return value / (1024 * 1024)

        return None

    def _format_seconds(self, value: Any) -> str:
        number = self._number(value)
        if number is None:
            return "-"
        return f"{number:.4f}s".rstrip("0").rstrip(".")


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
            drawing.add(String(0, y,name, fontSize=7, fillColor=colors.HexColor("#111827")))
            drawing.add(Rect(label_width, y - 3, bar_width, 6, fillColor=colors.HexColor("#E5E7EB"), strokeColor=None))
            drawing.add(Rect(label_width, y - 3, bar_width * max(0.0, value / max_value), 6, fillColor=colors.HexColor("#2563EB"), strokeColor=None))
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
    
    def _metric_table(self, metrics: Any, config: dict[str, Any],) -> tuple[list[str], list[list[Any]]]:
        rows = self._metric_rows(metrics, config)

        grouped: dict[str, dict[int, Any]] = {}
        ks: set[int] = set()

        for name, value in rows:
            metric, _, raw_k = name.partition("@")

            if not raw_k:
                continue

            digits = "".join(ch for ch in raw_k if ch.isdigit())
            if not digits:
                continue

            k = int(digits)
            grouped.setdefault(metric, {})[k] = value
            ks.add(k)

        ordered_metrics = ["NDCG", "MAP", "Recall", "Precision", "MRR"]
        ordered_ks = sorted(ks)

        headers = ["Metric"] + [f"@{k}" for k in ordered_ks]

        table_rows = []

        for metric in ordered_metrics:
            if metric not in grouped:
                continue

            row = [metric]

            for k in ordered_ks:
                value = grouped[metric].get(k)
                row.append("-" if value is None else round(value, 4))

            table_rows.append(row)

        return headers, table_rows

    def _metrics_to_columns(self, metrics: Any) -> dict[str, float]:

        result = {}

        metric_names = ("NDCG", "MAP", "Recall", "Precision")

        if isinstance(metrics, (list, tuple)) and len(metrics) == 4:

            for metric_name, metric_values in zip(metric_names, metrics):

                if not isinstance(metric_values, dict):
                    continue

                values = metric_values.get("summary", metric_values)

                for k, v in values.items():

                    number = self._number(v)

                    if number is not None:
                        result[f"{metric_name}@{k}"] = number

        return result

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

    def _config_rows(self, retriever: Any, config: dict[str, Any]) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = []
        seen: set[str] = set()

        def add(label: str, value: Any, keep_none: bool = False) -> None:
            if value is None and not keep_none:
                return
            if not self._is_scalar(value):
                return
            key = label.lower()
            if key in seen:
                return
            seen.add(key)
            rows.append((label, value))

        add("retriever_type", getattr(retriever, "retriever_type", type(retriever).__name__))
        for label, value in self._flatten(getattr(retriever, "config", {}) or {}):
            add(label, value, keep_none=True)

        for label in (
            "topk",
            "k",
            "language",
            "index_name",
            "content_field",
            "metadata_field",
            "refresh_on_write",
            "timeout",
            "model_id",
            "max_length",
            "device",
            "query_prompt_name",
            "passage_prompt_name",
            "normalize_embeddings",
            "batch_size",
            "corpus_chunk_size",
            "query_chunk_size",
            "show_progress_bar",
            "use_gpu_for_spacy",
        ):
            add(label, getattr(retriever, label, None))

        retrievers = getattr(retriever, "retrievers", None)
        if isinstance(retrievers, list) and retrievers:
            parts = []
            for child, weight in retrievers:
                child_name = getattr(child, "retriever_type", type(child).__name__) if not isinstance(child, dict) else "dict"
                parts.append(f"{child_name}:{self._format(weight)}")
            add("retrievers", ", ".join(parts))

        return rows[: self._int(config.get("config_rows", 30), 30)]

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
        return self._query_text(query, query_id)

    def _get_text(self, corpus: dict[Any, Any], doc_id: str) -> str:
        doc = self._lookup(corpus, doc_id)
        return self._doc_text(doc)

    def _query_text(self, query: Any, fallback: str = "") -> str:
        if isinstance(query, dict):
            return str(query.get("text") or query.get("query") or query.get("query_text") or fallback)
        return str(query or fallback)

    def _doc_text(self, doc: Any) -> str:
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
        styles.add(styles["BodyText"].clone("Intro", fontSize=9, leading=12, spaceAfter=4, textColor=colors.HexColor("#374151")))
        styles.add(styles["BodyText"].clone("Body", fontSize=9, leading=11, spaceAfter=4))
        styles.add(styles["BodyText"].clone("Small", fontSize=8, leading=10, spaceAfter=2))
        styles.add(styles["BodyText"].clone("Muted", fontSize=8, leading=10, textColor=colors.HexColor("#6B7280")))
        styles.add(styles["BodyText"].clone("Cell", fontSize=8, leading=10))
        styles.add(styles["BodyText"].clone("HeaderCell", fontSize=8, leading=10, textColor=colors.white))
        styles.add(styles["BodyText"].clone("Note", fontSize=8, leading=10, textColor=colors.HexColor("#374151"), leftIndent=4, rightIndent=4))
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

    def _ordered_metric_names(self, names: set[str]) -> list[str]:
        order = {"NDCG": 0, "MAP": 1, "Recall": 2, "Precision": 3, "MRR": 4}

        def key(name: str) -> tuple[int, int, str]:
            metric, _, raw_k = name.partition("@")
            digits = "".join(ch for ch in raw_k if ch.isdigit())
            return (order.get(metric, len(order)), int(digits or 0), name)

        return sorted(names, key=key)

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

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _is_scalar(self, value: Any) -> bool:
        return isinstance(value, (str, int, float, bool)) or value is None

    def _esc(self, value: Any) -> str:
        return escape(str(value or ""))
