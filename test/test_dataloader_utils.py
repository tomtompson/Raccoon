import tempfile
import unittest
from pathlib import Path

from raccoon.dataloader.utils import (
    load_synthesized_results,
    save_synthesized_results,
)


class SynthesizedResultsUtilsTest(unittest.TestCase):
    def test_save_and_load_synthesized_results_round_trip(self) -> None:
        rows = [
            {
                "passage": "A Zeliox unit contains lithium cells.",
                "question": "What type of cells does a Zeliox unit contain?",
                "answer": "lithium cells",
                "metadata": {"source": "unit", "page": 1},
                "groundedness_score": 5,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generated_qa.jsonl"
            save_synthesized_results(rows, output_path)
            loaded_rows = load_synthesized_results(output_path)

        self.assertEqual(loaded_rows, rows)


if __name__ == "__main__":
    unittest.main()
