from __future__ import annotations

from pathlib import Path
from typing import Any

from fpdf import FPDF

from raccoon.eval.retriever.RetrievelEval import RetrievelEval
from raccoon.retriever.BaseRetriever import BaseRetriever


class StaticSingleRetrieverReport:
    def __init__(
        self,
        title: str | None = None,
        retriever: BaseRetriever | None = None,
        retriever_eval: RetrievelEval | None = None,
        raw_eval: dict[str, Any] | None = None,
        raw_results: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.title = title
        self.retriever = retriever
        self.retriever_eval = retriever_eval
        self.raw_eval = raw_eval
        self.raw_results = raw_results

        self.summary_results: dict[str, float] = {}
        self.diagnostics_results: dict[str, Any] = {}
        self.per_query_results: dict[str, dict[str, float]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.queries: dict[str, dict[str, Any]] = {}
        self.qrels: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.retriever_metrics: dict[str, Any] = {}

    def generate_report(
        self,
        path: Path | str = "data/processed/",
        name: str = "Retriever Report",
    ) -> Path:
        self._load_inputs()

        output_path = Path(path)
        if output_path.suffix.lower() == ".pdf":
            target_path = output_path
        else:
            if Path(name).suffix.lower() != ".pdf":
                name = name + ".pdf"
            target_path = output_path / name 
        target_path.parent.mkdir(parents=True, exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        self._render_header(pdf)
        self._render_diagnostics(pdf)
        self._render_retriever_metrics(pdf)
        self._render_metrics(pdf)
        self._render_low_score_queries(pdf)

        pdf.output(str(target_path))
        return target_path


    def _load_inputs(self) -> None:
        eval_payload = self.raw_eval
        if eval_payload is None:
            if self.retriever_eval is None:
                raise ValueError("Provide raw_eval or retriever_eval.")
            else:
                eval_payload = self.retriever_eval.raw_results

        results_payload = self.raw_results
        if results_payload is None:
            if self.retriever is not None:
                results_payload = getattr(self.retriever, "benchmark_data", {}) or {}
            elif self.retriever_eval is not None and self.retriever_eval.retriever is not None:
                results_payload = getattr(self.retriever_eval.retriever, "benchmark_data", {}) or {}
            else:
                raise ValueError("Provide raw_results or a retriever with benchmark_data.")
            
        

        self.summary_results = dict(eval_payload.get("summary", {}) or {})
        self.diagnostics_results = dict(eval_payload.get("diagnostics", {}) or {})
        self.per_query_results = dict(eval_payload.get("per_query", {}) or {})

        self.documents = dict(results_payload.get("documents", {}) or {})
        self.queries = dict(results_payload.get("queries", {}) or {})
        self.qrels = dict(results_payload.get("qrels", {}) or {})
        self.results = dict(results_payload.get("results", {}) or {})
        if self.retriever is not None:
            self.retriever_metrics = dict(getattr(self.retriever, "metrics", {}) or {})
        elif self.retriever_eval is not None and self.retriever_eval.retriever is not None:
            self.retriever_metrics = dict(getattr(self.retriever_eval.retriever, "metrics", {}) or {})
        else:
            self.retriever_metrics = {}

        self.diagnostics_results["document_count"] = len(self.documents)
    def _top_k_values(self) -> list[int]:
        ks: set[int] = set()
        for name in self.summary_results:
            if "@" not in name:
                continue
            _, raw_k = name.split("@", 1)
            if raw_k.isdigit():
                ks.add(int(raw_k))
        return sorted(ks)

    def _group_summary_metrics(self) -> dict[str, list[tuple[int, float]]]:
        grouped: dict[str, list[tuple[int, float]]] = {}
        for name, value in self.summary_results.items():
            if "@" not in name:
                continue
            metric_name, raw_k = name.split("@", 1)
            if not raw_k.isdigit():
                continue
            grouped.setdefault(metric_name, []).append((int(raw_k), float(value)))

        for metric_name in grouped:
            grouped[metric_name].sort(key=lambda item: item[0])
        return grouped

    def _ensure_space(self, pdf: FPDF, required_height: float) -> None:
        if pdf.get_y() + required_height <= 280:
            return
        pdf.add_page()

    def _section_title(self, pdf: FPDF, title: str) -> None:
        self._ensure_space(pdf, 12)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, self._pdf_text(title), ln=1)
        pdf.ln(1)

    def _draw_metric_bar(self, pdf: FPDF, label: str, value: float) -> None:
        self._ensure_space(pdf, 10)
        bar_x = 78
        bar_y = pdf.get_y() + 1
        bar_width = 95
        bar_height = 5
        normalized = max(0.0, min(float(value), 1.0))

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(45, 7, self._pdf_text(label))
        pdf.cell(25, 7, self._pdf_text(f"{normalized:.3f}"))
        pdf.set_draw_color(190, 190, 190)
        pdf.rect(bar_x, bar_y, bar_width, bar_height)
        pdf.set_fill_color(52, 101, 164)
        pdf.rect(bar_x, bar_y, bar_width * normalized, bar_height, style="F")
        pdf.ln(7)

    def _truncate(self, text: str, limit: int = 280) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _pdf_text(self, text: Any) -> str:
        return str(text or "").encode("latin-1", "replace").decode("latin-1")

    def _get_query_text(self, query_id: str) -> str:
        payload = self.queries.get(query_id, {})
        return str(payload.get("text") or payload.get("query_text") or query_id)

    def _relevant_document_ids(self, query_id: str) -> set[str]:
        relevant_ids: set[str] = set()
        for qrel in self.qrels.values():
            if str(qrel.get("query_id")) != str(query_id):
                continue
            document_id = str(qrel.get("document_id", "")).strip()
            if document_id:
                relevant_ids.add(document_id)
        return relevant_ids

    def _document_text(self, document_id: str, hit: dict[str, Any] | None = None) -> str:
        document = self.documents.get(document_id, {})
        if document.get("text"):
            return str(document["text"])
        if hit and hit.get("content"):
            return str(hit["content"])
        return ""

    def _query_score(self, query_id: str) -> tuple[float, float, float]:
        per_query = self.per_query_results.get(query_id, {})
        ks = self._top_k_values()
        max_k = max(ks) if ks else 1
        recall = float(per_query.get(f"recall_{max_k}", 0.0))
        precision = float(per_query.get(f"P_{max_k}", 0.0))
        smaller_k = min(ks) if ks else 1
        early_precision = float(per_query.get(f"P_{smaller_k}", 0.0))
        return (recall, precision, early_precision)

    def _low_score_queries(self, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query_id in self.per_query_results:
            result = self.results.get(query_id, {})
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": self._get_query_text(query_id),
                    "scores": self._query_score(query_id),
                    "hits": list(result.get("hits", [])),
                    "relevant_document_ids": self._relevant_document_ids(query_id),
                }
            )

        rows.sort(key=lambda item: (item["scores"][0], item["scores"][1], item["scores"][2], item["query_id"]))
        return rows[:limit]

    def _render_header(self, pdf: FPDF) -> None:
        pdf.set_font("Helvetica", "B", 18)
        if self.title:
            pdf.cell(0,10, self.title)
        else:
            pdf.cell(0, 10, f"Retriever Report {type(self.retriever).__name__ if self.retriever else ""}", ln=1)
        pdf.image(name="raccoon/report/images/logo.png", x = 160, y = -5, w = 50  )
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(10)
        evaluated = self.diagnostics_results.get("queries_evaluated", 0)
        total = self.diagnostics_results.get("queries_total", 0)
        pdf.cell(0, 6, f"Evaluated queries: {evaluated}/{total}", ln=1)
        pdf.ln(4)

    def _render_diagnostics(self, pdf: FPDF) -> None:
        self._section_title(pdf, "Diagnostics")
        pdf.set_font("Helvetica", "", 10)
        for key in self.diagnostics_results.keys():
            pdf.cell(65, 6, key.replace("_", " ").title())
            pdf.cell(0, 6, str(self.diagnostics_results[key]), ln=1)
        pdf.ln(2)

    def _render_metrics(self, pdf: FPDF) -> None:
        self._section_title(pdf, "Summary Metrics")
        for metric_name, values in self._group_summary_metrics().items():
            self._ensure_space(pdf, 50)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, str(metric_name), ln=1)
            for k, value in values:
                self._draw_metric_bar(pdf, f"{metric_name}@{k}", value)
            pdf.ln(1)
        pdf.ln(5)

    def _render_retriever_metrics(self, pdf: FPDF) -> None:
        if not self.retriever_metrics:
            return

        self._section_title(pdf, "Retriever Metrics")
        pdf.set_font("Helvetica", "", 10)
        for key, value in self.retriever_metrics.items():
            self._ensure_space(pdf, 8)
            pdf.cell(65, 6, self._pdf_text(key.replace("_", " ").title()))
            pdf.cell(0, 6, self._pdf_text(value), ln=1)
        pdf.ln(2)

    def _render_low_score_queries(self, pdf: FPDF) -> None:
        sample_size = int(self.config.get("low_score_examples", 5))
        low_score_queries = self._low_score_queries(limit=sample_size)
        if not low_score_queries:
            return

        self._section_title(pdf, "Lower Score Queries")
        ks = self._top_k_values()
        max_k = max(ks) if ks else 1
        available_width = pdf.w - pdf.l_margin - pdf.r_margin

        for index, row in enumerate(low_score_queries, start=1):
            self._ensure_space(pdf, 40)
            recall, precision, early_precision = row["scores"]
            pdf.set_font("Helvetica", "B", 11)
            text = self._pdf_text(f"{index}. {str(row['query_text'])[:200]}")
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                available_width,
                6,
                text,
            )
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                available_width,
                5,
                self._pdf_text(
                    f"Recall@{max_k}: {recall:.3f} | "
                    f"P@{max_k}: {precision:.3f} | "
                    f"Early precision: {early_precision:.3f}"
                ),
            )

            top_hits = row["hits"][: int(self.config.get("documents_per_query", 3))]
            if not top_hits:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(available_width, 5, self._pdf_text("No retrieved documents."))
                pdf.ln(2)
                continue

            for hit in top_hits:
                self._ensure_space(pdf, 20)
                document_id = str(hit.get("document_id", ""))
                is_relevant = document_id in row["relevant_document_ids"]
                label = "relevant" if is_relevant else "not relevant"
                snippet = self._truncate(self._document_text(document_id, hit), 220) if not is_relevant else self._document_text(document_id, hit),
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    available_width,
                    5,
                    self._pdf_text(
                        f"Doc {hit.get('rank', '?')} | score={float(hit.get('score', 0.0)):.3f} | {label} | {document_id}"
                    ),
                )
                pdf.set_font("Helvetica", "", 9)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(available_width, 5, self._pdf_text(snippet or "(no document text available)"))
                pdf.ln(1)

            pdf.ln(2)
