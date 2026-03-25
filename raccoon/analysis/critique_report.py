from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_THRESHOLDS = {
    "groundedness": 4,
    "relevance": 4,
    "standalone": 4,
}


@dataclass(frozen=True)
class CritiqueSummary:
    total_rows: int
    review_required: int
    average_total_score: float
    median_total_score: float
    score_distributions: dict[str, dict[int, int]]


def load_critique_rows(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_critique_row(
    row: dict[str, Any],
    thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    normalized = dict(row)
    metadata = dict(normalized.get("metadata") or {})

    question = _clean_text(normalized.get("question", ""))
    answer = _clean_text(normalized.get("answer", ""))
    generated = _clean_text(normalized.get("generated", ""))
    passage = _clean_text(normalized.get("passage", normalized.get("context", "")))

    groundedness = _safe_int(normalized.get("groundedness_score"))
    relevance = _safe_int(normalized.get("relevance_score"))
    standalone = _safe_int(normalized.get("standalone_score"))

    low_groundedness = groundedness < thresholds["groundedness"]
    low_relevance = relevance < thresholds["relevance"]
    low_standalone = standalone < thresholds["standalone"]
    context_dependent = low_standalone or _contains_context_reference(question)
    likely_trivia = low_relevance and not context_dependent

    normalized.update(
        {
            "question": question,
            "answer": answer,
            "generated": generated,
            "passage": passage,
            "metadata": metadata,
            "source": metadata.get("source", ""),
            "page": metadata.get("page", ""),
            "page_label": metadata.get("page_label", ""),
            "groundedness_score": groundedness,
            "relevance_score": relevance,
            "standalone_score": standalone,
            "total_score": groundedness + relevance + standalone,
            "low_groundedness": low_groundedness,
            "low_relevance": low_relevance,
            "low_standalone": low_standalone,
            "context_dependent": context_dependent,
            "likely_trivia": likely_trivia,
            "review_required": low_groundedness or low_relevance or low_standalone,
        }
    )
    return normalized


def build_critique_report(
    input_path: str | Path,
    output_dir: str | Path,
    thresholds: dict[str, int] | None = None,
) -> dict[str, Path | str | CritiqueSummary]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = [normalize_critique_row(row, thresholds) for row in load_critique_rows(input_path)]
    summary = summarize_rows(rows)

    normalized_csv = output_path / "critique_report.csv"
    flagged_csv = output_path / "critique_flagged.csv"
    summary_txt = output_path / "summary.txt"

    write_csv(rows, normalized_csv)
    write_csv([row for row in rows if row["review_required"]], flagged_csv)
    summary_txt.write_text(render_summary(summary, rows), encoding="utf-8")
    save_charts(rows, output_path)

    return {
        "rows": rows,
        "summary": summary,
        "normalized_csv": normalized_csv,
        "flagged_csv": flagged_csv,
        "summary_txt": summary_txt,
        "charts_dir": output_path,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> CritiqueSummary:
    totals = [row["total_score"] for row in rows]
    score_distributions = {
        criterion: dict(sorted(Counter(row[f"{criterion}_score"] for row in rows).items()))
        for criterion in ("groundedness", "relevance", "standalone")
    }
    return CritiqueSummary(
        total_rows=len(rows),
        review_required=sum(1 for row in rows if row["review_required"]),
        average_total_score=round(mean(totals), 2) if totals else 0.0,
        median_total_score=float(median(totals)) if totals else 0.0,
        score_distributions=score_distributions,
    )


def render_summary(summary: CritiqueSummary, rows: list[dict[str, Any]]) -> str:
    worst_rows = sorted(
        rows,
        key=lambda row: (row["total_score"], row["relevance_score"], row["standalone_score"]),
    )[:5]
    disagreement_rows = sorted(
        rows,
        key=lambda row: abs(row["groundedness_score"] - row["relevance_score"]),
        reverse=True,
    )[:5]

    lines = [
        "Critique report",
        f"Rows: {summary.total_rows}",
        f"Review required: {summary.review_required}",
        f"Average total score: {summary.average_total_score}",
        f"Median total score: {summary.median_total_score}",
        "",
        "Score distributions:",
    ]
    for criterion, distribution in summary.score_distributions.items():
        histogram = ", ".join(f"{score}:{count}" for score, count in distribution.items())
        lines.append(f"- {criterion}: {histogram}")

    lines.append("")
    lines.append("Worst examples:")
    for row in worst_rows:
        lines.append(
            f"- total={row['total_score']} page={row['page']} "
            f"g={row['groundedness_score']} r={row['relevance_score']} s={row['standalone_score']} "
            f"question={row['question']}"
        )

    lines.append("")
    lines.append("Largest groundedness vs relevance gaps:")
    for row in disagreement_rows:
        lines.append(
            f"- gap={abs(row['groundedness_score'] - row['relevance_score'])} "
            f"page={row['page']} g={row['groundedness_score']} r={row['relevance_score']} "
            f"question={row['question']}"
        )

    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    fieldnames = [
        "question",
        "answer",
        "source",
        "page",
        "page_label",
        "groundedness_score",
        "relevance_score",
        "standalone_score",
        "total_score",
        "low_groundedness",
        "low_relevance",
        "low_standalone",
        "context_dependent",
        "likely_trivia",
        "review_required",
        "groundedness_eval",
        "relevance_eval",
        "standalone_eval",
        "generated",
        "passage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_charts(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    _plot_histograms(rows, output_path / "score_histograms.svg")
    _plot_pass_fail(rows, output_path / "flag_summary.svg")
    _plot_score_scatter(rows, output_path / "groundedness_vs_relevance.svg")


def _plot_histograms(rows: list[dict[str, Any]], output_path: Path) -> None:
    sections: list[str] = []
    for index, (criterion, color) in enumerate(
        zip(
            ("groundedness", "relevance", "standalone"),
            ("#1f77b4", "#ff7f0e", "#2ca02c"),
        )
    ):
        distribution = Counter(row[f"{criterion}_score"] for row in rows)
        title = f"{criterion.title()} score"
        sections.append(
            _bar_chart_svg(
                x=20 + index * 300,
                y=50,
                width=240,
                height=220,
                labels=[str(score) for score in range(1, 6)],
                values=[distribution.get(score, 0) for score in range(1, 6)],
                color=color,
                title=title,
            )
        )

    _write_svg(output_path, 920, 320, "\n".join(sections))


def _plot_pass_fail(rows: list[dict[str, Any]], output_path: Path) -> None:
    criteria = ["groundedness", "relevance", "standalone"]
    passed = [sum(not row[f"low_{criterion}"] for row in rows) for criterion in criteria]
    failed = [sum(row[f"low_{criterion}"] for row in rows) for criterion in criteria]
    max_value = max((pass_count + fail_count for pass_count, fail_count in zip(passed, failed)), default=1)
    parts = [
        '<text x="20" y="24" font-size="18" font-family="sans-serif">Threshold outcomes by criterion</text>',
        _legend_svg(580, 40, [("Pass", "#4daf4a"), ("Review", "#e41a1c")]),
    ]
    chart_width = 150
    for index, criterion in enumerate(criteria):
        x = 40 + index * 180
        total_height = 180
        passed_height = 0 if max_value == 0 else total_height * (passed[index] / max_value)
        failed_height = 0 if max_value == 0 else total_height * (failed[index] / max_value)
        base_y = 250
        parts.append(f'<rect x="{x}" y="{base_y - passed_height}" width="{chart_width}" height="{passed_height}" fill="#4daf4a" />')
        parts.append(f'<rect x="{x}" y="{base_y - passed_height - failed_height}" width="{chart_width}" height="{failed_height}" fill="#e41a1c" />')
        parts.append(f'<text x="{x + (chart_width / 2)}" y="275" text-anchor="middle" font-size="12" font-family="sans-serif">{escape(criterion)}</text>')
        parts.append(f'<text x="{x + (chart_width / 2)}" y="{base_y - passed_height - failed_height - 8}" text-anchor="middle" font-size="11" font-family="sans-serif">{passed[index] + failed[index]}</text>')

    _write_svg(output_path, 760, 320, "\n".join(parts))


def _plot_score_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    width = 420
    height = 280
    chart_x = 70
    chart_y = 30
    parts = [
        '<text x="20" y="24" font-size="18" font-family="sans-serif">Groundedness vs relevance</text>',
        _legend_svg(380, 20, [("Pass", "#377eb8"), ("Review", "#e41a1c")]),
        f'<rect x="{chart_x}" y="{chart_y}" width="{width}" height="{height}" fill="white" stroke="#333" />',
    ]

    for tick in range(1, 6):
        px = chart_x + ((tick - 1) / 4) * width
        py = chart_y + height - ((tick - 1) / 4) * height
        parts.append(f'<line x1="{px}" y1="{chart_y}" x2="{px}" y2="{chart_y + height}" stroke="#ddd" />')
        parts.append(f'<line x1="{chart_x}" y1="{py}" x2="{chart_x + width}" y2="{py}" stroke="#ddd" />')
        parts.append(f'<text x="{px}" y="{chart_y + height + 18}" text-anchor="middle" font-size="12" font-family="sans-serif">{tick}</text>')
        parts.append(f'<text x="{chart_x - 18}" y="{py + 4}" text-anchor="middle" font-size="12" font-family="sans-serif">{tick}</text>')

    parts.append(f'<text x="{chart_x + (width / 2)}" y="{chart_y + height + 38}" text-anchor="middle" font-size="12" font-family="sans-serif">Groundedness</text>')
    parts.append(f'<text x="18" y="{chart_y + (height / 2)}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 18 {chart_y + (height / 2)})">Relevance</text>')

    for row in rows:
        cx = chart_x + ((row["groundedness_score"] - 1) / 4) * width
        cy = chart_y + height - ((row["relevance_score"] - 1) / 4) * height
        color = "#e41a1c" if row["review_required"] else "#377eb8"
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{color}" fill-opacity="0.75" />')

    _write_svg(output_path, 560, 380, "\n".join(parts))



def _contains_context_reference(question: str) -> bool:
    lowered = question.lower()
    if "diagram" in lowered or "figure" in lowered:
        return True
    markers = (
        "this document",
        "the document",
        "this passage",
        "the passage",
        "this context",
        "the context",
        "according to",
        "in the diagram",
        "in the figure",
        "from the diagram",
        "from the figure",
    )
    return any(marker in lowered for marker in markers)


def build_report(
    input_path: str | Path,
    output_dir: str | Path,
    thresholds: dict[str, int] | None = None,
) -> dict[str, Path | str | CritiqueSummary]:
    return build_critique_report(input_path, output_dir, thresholds)


def _clean_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def _safe_int(value: Any) -> int:
    return int(value) if value is not None else 0


def _write_svg(output_path: Path, width: int, height: int, body: str) -> None:
    output_path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            f'<rect width="100%" height="100%" fill="white" />\n{body}\n</svg>\n'
        ),
        encoding="utf-8",
    )


def _bar_chart_svg(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    labels: list[str],
    values: list[float],
    color: str,
    title: str,
    rotate_labels: bool = False,
    value_decimals: int = 0,
) -> str:
    max_value = max(values, default=1) or 1
    bar_width = width / max(len(values), 1)
    parts = [
        f'<text x="{x}" y="{y - 16}" font-size="18" font-family="sans-serif">{escape(title)}</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white" stroke="#333" />',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = 0 if max_value == 0 else (value / max_value) * (height - 30)
        bar_x = x + index * bar_width + 8
        bar_y = y + height - bar_height - 20
        inner_width = max(bar_width - 16, 8)
        value_label = f"{value:.{value_decimals}f}" if value_decimals else f"{int(value)}"
        parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="{inner_width}" height="{bar_height}" fill="{color}" />')
        parts.append(f'<text x="{bar_x + (inner_width / 2)}" y="{bar_y - 6}" text-anchor="middle" font-size="11" font-family="sans-serif">{escape(value_label)}</text>')

        label_x = bar_x + (inner_width / 2)
        label_y = y + height - 4
        if rotate_labels:
            parts.append(
                f'<text x="{label_x}" y="{label_y}" text-anchor="end" font-size="11" '
                f'font-family="sans-serif" transform="rotate(-45 {label_x} {label_y})">{escape(label)}</text>'
            )
        else:
            parts.append(f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="11" font-family="sans-serif">{escape(label)}</text>')
    return "\n".join(parts)


def _legend_svg(x: int, y: int, items: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for index, (label, color) in enumerate(items):
        item_y = y + index * 18
        parts.append(f'<rect x="{x}" y="{item_y - 10}" width="12" height="12" fill="{color}" />')
        parts.append(f'<text x="{x + 18}" y="{item_y}" font-size="12" font-family="sans-serif">{escape(label)}</text>')
    return "\n".join(parts)
