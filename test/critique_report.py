import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crimsonvector.analysis import build_critique_report


INPUT_PATH = Path("data/processed/critique.jsonl")
OUTPUT_DIR = Path("data/processed/critique_report")


def main() -> None:
    report = build_critique_report(INPUT_PATH, OUTPUT_DIR)
    summary = report["summary"]

    print(f"Saved report CSV to {report['normalized_csv']}")
    print(f"Saved flagged CSV to {report['flagged_csv']}")
    print(f"Saved summary to {report['summary_txt']}")
    print(f"Rows: {summary.total_rows}")
    print(f"Review required: {summary.review_required}")
    print(f"Average total score: {summary.average_total_score}")


if __name__ == "__main__":
    main()
