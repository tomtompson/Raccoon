import csv
import json
import tempfile
import unittest
from pathlib import Path

from raccoon.analysis.critique_report import build_critique_report, normalize_critique_row


class CritiqueReportTest(unittest.TestCase):
    def test_normalize_critique_row_trims_fields_and_sets_flags(self) -> None:
        row = {
            "question": "  What kind of background isn't necessary to understand this document?\n",
            "answer": " Mathematical background. ",
            "generated": " Factoid question: test ",
            "passage": " Example passage. ",
            "metadata": {"source": "data/raw/Quicmaffs.pdf", "page": 2, "page_label": "3"},
            "groundedness_score": 5,
            "relevance_score": 1,
            "standalone_score": 1,
        }

        normalized = normalize_critique_row(row)

        self.assertEqual(
            normalized["question"],
            "What kind of background isn't necessary to understand this document?",
        )
        self.assertEqual(normalized["answer"], "Mathematical background.")
        self.assertEqual(normalized["source"], "data/raw/Quicmaffs.pdf")
        self.assertEqual(normalized["total_score"], 7)
        self.assertTrue(normalized["low_relevance"])
        self.assertTrue(normalized["low_standalone"])
        self.assertTrue(normalized["context_dependent"])
        self.assertTrue(normalized["review_required"])
        self.assertFalse(normalized["likely_trivia"])

    def test_build_critique_report_writes_expected_outputs(self) -> None:
        rows = [
            {
                "question": "What is Bayes' theorem?",
                "answer": "A formula for updating beliefs.",
                "generated": "Factoid question: What is Bayes' theorem? Answer: A formula for updating beliefs.",
                "passage": "Bayes theorem updates beliefs with evidence.",
                "metadata": {"source": "doc.pdf", "page": 1, "page_label": "2"},
                "groundedness_score": 5,
                "relevance_score": 5,
                "standalone_score": 5,
                "groundedness_eval": "Strongly supported.",
                "relevance_eval": "Useful.",
                "standalone_eval": "Independent.",
            },
            {
                "question": "What kind of background isn't necessary to understand this document?",
                "answer": "Mathematical background.",
                "generated": "Factoid question: What kind of background isn't necessary to understand this document? Answer: Mathematical background.",
                "passage": "The document is written to be understood without mathematics.",
                "metadata": {"source": "doc.pdf", "page": 2, "page_label": "3"},
                "groundedness_score": 5,
                "relevance_score": 1,
                "standalone_score": 1,
                "groundedness_eval": "Supported.",
                "relevance_eval": "Weak.",
                "standalone_eval": "Context dependent.",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "critique.jsonl"
            output_dir = Path(tmp_dir) / "report"

            with input_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            report = build_critique_report(input_path, output_dir)

            self.assertTrue(Path(report["normalized_csv"]).exists())
            self.assertTrue(Path(report["flagged_csv"]).exists())
            self.assertTrue(Path(report["summary_txt"]).exists())
            self.assertTrue((output_dir / "score_histograms.svg").exists())
            self.assertTrue((output_dir / "flag_summary.svg").exists())
            self.assertTrue((output_dir / "groundedness_vs_relevance.svg").exists())

            with Path(report["flagged_csv"]).open("r", encoding="utf-8") as handle:
                flagged_rows = list(csv.DictReader(handle))

            self.assertEqual(len(flagged_rows), 1)
            self.assertEqual(flagged_rows[0]["question"], rows[1]["question"])

            summary_text = Path(report["summary_txt"]).read_text(encoding="utf-8")
            self.assertIn("Rows: 2", summary_text)
            self.assertIn("Review required: 1", summary_text)

    def test_normalize_critique_row_marks_diagram_reference_as_context_dependent(self) -> None:
        row = {
            "question": "What does 'G' represent in the data server diagram?",
            "answer": "graph structure",
            "generated": "Factoid question: What does 'G' represent in the data server diagram? Answer: graph structure",
            "passage": "G= graph structure",
            "metadata": {"source": "doc.pdf", "page": 12, "page_label": "13"},
            "groundedness_score": 5,
            "relevance_score": 2,
            "standalone_score": 4,
        }

        normalized = normalize_critique_row(row)

        self.assertTrue(normalized["context_dependent"])
        self.assertFalse(normalized["likely_trivia"])


if __name__ == "__main__":
    unittest.main()
